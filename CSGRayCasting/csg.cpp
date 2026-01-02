#include "csg.h"
#include <algorithm>

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
    if (shape != nullptr) {
        // Leaf node: delegate to the primitive Shape
        return shape->getSpans(ray);
    }
    else if (left != nullptr && right != nullptr) {
        // Internal node: recursively get spans from children and combine
        auto left_spans = left->getSpans(ray);
        auto right_spans = right->getSpans(ray);
        return combineSpans(left_spans, right_spans, op);
    }
    else {
        // Invalid node configuration: return empty spans
        return {};
    }
}