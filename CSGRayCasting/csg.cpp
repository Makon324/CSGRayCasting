#include "csg.h"
#include <algorithm>
#include <stack>

struct Event {
    float t;
    bool is_entry;
    bool is_left;  // true for left subtree, false for right
    const Hit* hit;
    bool operator<(const Event& other) const {
        if (t != other.t) return t < other.t;
        return is_entry > other.is_entry;  // Entries before exits at same t
    }
};

std::vector<Span> combineSpans(const std::vector<Span>& left_spans, const std::vector<Span>& right_spans, CSGOp op) {
    std::vector<Event> events;
    for (const auto& s : left_spans) {
        events.push_back({ s.t_entry, true, true, &s.entry_hit });
        events.push_back({ s.t_exit, false, true, &s.exit_hit });
    }
    for (const auto& s : right_spans) {
        events.push_back({ s.t_entry, true, false, &s.entry_hit });
        events.push_back({ s.t_exit, false, false, &s.exit_hit });
    }
    std::sort(events.begin(), events.end());

    std::vector<Span> result;
    int count_left = 0, count_right = 0;
    float start_t = -1;
    Hit start_hit;
    bool prev_inside = false;

    for (const auto& e : events) {
        bool curr_inside = false;
        if (e.is_left) {
            if (e.is_entry) ++count_left;
            else --count_left;
        }
        else {
            if (e.is_entry) ++count_right;
            else --count_right;
        }

        switch (op) {
        case CSGOp::UNION:
            curr_inside = (count_left > 0 || count_right > 0);
            break;
        case CSGOp::INTERSECTION:
            curr_inside = (count_left > 0 && count_right > 0);
            break;
        case CSGOp::DIFFERENCE:
            curr_inside = (count_left > 0 && count_right == 0);
            break;
        }

        if (!prev_inside && curr_inside) {
            // Start new span
            start_t = e.t;
            start_hit = *e.hit;
            if (op == CSGOp::DIFFERENCE && !e.is_left) {
                start_hit.normal = start_hit.normal * -1.0f;  // Flip for subtrahend surface
            }
        }
        else if (prev_inside && !curr_inside) {
            // End span
            Span s;
            s.t_entry = start_t;
            s.entry_hit = start_hit;
            s.t_exit = e.t;
            s.exit_hit = *e.hit;
            if (op == CSGOp::DIFFERENCE && !e.is_left) {
                s.exit_hit.normal = s.exit_hit.normal * -1.0f;  // Flip
            }
            result.push_back(s);
            start_t = -1;
        }
        prev_inside = curr_inside;
    }
    return result;
}

std::vector<Span> CSGNode::getSpans(const Ray& ray) const {
    if (this == nullptr) {
        return {};
    }

    std::stack<const CSGNode*> s1, s2;
    s1.push(this);

    while (!s1.empty()) {
        const CSGNode* curr = s1.top();
        s1.pop();
        s2.push(curr);

        if (curr->left != nullptr) {
            s1.push(curr->left);
        }
        if (curr->right != nullptr) {
            s1.push(curr->right);
        }
    }

    std::stack<std::vector<Span>> value_stack;

    while (!s2.empty()) {
        const CSGNode* curr = s2.top();
        s2.pop();

        std::vector<Span> spans;
        if (curr->shape != nullptr) {
            // Leaf node: delegate to the primitive Shape
            spans = curr->shape->getSpans(ray);
        }
        else if (curr->left != nullptr && curr->right != nullptr) {
            // Internal node: combine spans from children
            std::vector<Span> right_spans = value_stack.top();
            value_stack.pop();
            std::vector<Span> left_spans = value_stack.top();
            value_stack.pop();
            spans = combineSpans(left_spans, right_spans, curr->op);
        }
        else {
            // Invalid node configuration: return empty spans
            spans = {};
        }
        value_stack.push(spans);
    }

    if (value_stack.empty()) {
        return {};
    }
    return value_stack.top();
}